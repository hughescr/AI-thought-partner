// eslint-disable-next-line n/no-unpublished-import -- This import is not published cos it's dev only
import defaultConfig from '@hughescr/eslint-config-default';

export default
[
    {
        name: 'ignores',
        ignores: ['coverage', 'node_modules'],
    },
    defaultConfig.configs.recommended,
];
